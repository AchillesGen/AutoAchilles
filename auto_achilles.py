
import os
import numpy as np
import matplotlib.pyplot as plt
import pyNUISANCE as pn
import re
from pathlib import Path
from string import Template
import subprocess

 #PDG ion → tag (e.g., 1000180400 → "Ar40")
SYMBOL_BY_Z = {
    1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
    11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",
    21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",
    31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",
    41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",
    51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",
    61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",
    71:"Lu",72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",
    81:"Tl",82:"Pb",83:"Bi",84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",
    91:"Pa",92:"U",93:"Np",94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",99:"Es",100:"Fm",
    101:"Md",102:"No",103:"Lr",104:"Rf",105:"Db",106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",
    111:"Rg",112:"Cn",113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og"
}

class AutoAchilles:
    '''
    A class for automating ACHILLES.
    
    Attributes:
        achilles_path (str): The local path to ACHILLES.
    '''
    def __init__(self, achilles_path):
        '''
        Initializes AutoAchilles.
        
        Parameters:
            achilles_path (str): The local path to ACHILLES.
            '''
        self.achilles_path = achilles_path
        self.options = {"run_cascade": False}
    
    def cascade_options(self, run_cascade):
        '''
        Runs cascading if true.
        
        Parameters:
            run_cascade (str): Can be set to True or False.
            '''
        self.options["run_cascade"] = run_cascade

    @staticmethod
    def pdg_ion_to_tag(code: str) -> str:
        """
        Accepts a 10-digit PDG ion code string (10LZZZAAAI) and returns 'SymA' like 'Ar40'.
        """
        s = str(code).strip()
        if not re.fullmatch(r"\d{10}", s) or not s.startswith("10"):
            raise ValueError(f"Not a PDG ion code: {code!r}")
        Z = int(s[3:6])
        A = int(s[6:9])
        symbol = SYMBOL_BY_Z.get(Z, f"Z{Z}")
        return f"{symbol}{A}"
        
    def run(self, record_ref = "hepdata-sandbox:1713917002", events = 10000):
        """
        Runs ACHILLES.
        
        Parameters:
            record_ref (str): The run card for ACHILLES.
            events (int): The amount of events to run.
        """
        rf = pn.RecordFactory()
        self.hepdata_rec = rf.make_record({"type":"hepdata",
            "recordref":record_ref})
        
        self.options |= {
            'probe_pdg': self.hepdata_rec.analysis(f"{self.hepdata_rec.get_analyses()[0]}").get_probe_flux(False).probe_pdg, # Probe PDG
            'flux_file': self.hepdata_rec.analysis(f"{self.hepdata_rec.get_analyses()[0]}").get_probe_flux(False).source, # Neutrino flux file
            'target_pdg': AutoAchilles.pdg_ion_to_tag(str(self.hepdata_rec.analysis(f"{self.hepdata_rec.get_analyses()[0]}").get_target()[0])), # Target PDG
            'events': events
        }

        with open('run_template.yml', 'r') as f:
            src = Template(f.read())
            result = src.substitute(self.options)
            outname = f"run_{record_ref.replace(":", "-")}.yml"

        with open(outname, 'w') as f:
            f.write(result)

        subprocess.run([self.achilles_path, outname],)

        self.achilles_events = pn.EventSource("achilles.hepmc")
        if not self.achilles_events:
            print("Failed to read file")
    
    def list_analyses(self):
        """
        Lists the different possible analyses.
        """
        return self.hepdata_rec.get_analyses()

    def plot(self, analysis_str):
        """
        Plots an analysis.
        
        Parameters:
            analysis_str (str): Which analysis to plot
        """
        analysis = self.hepdata_rec.analysis(f"{analysis_str}")
        comparison = analysis.process(self.achilles_events)
        self.make_plot(analysis, comparison)

    def plot_all(self):
        """
        Makes all the plots.
        """
        for analysis_str in self.hepdata_rec.get_analyses():
            analysis = self.hepdata_rec.analysis(f"{analysis_str}")
            comparison = analysis.process(self.achilles_events)
            self.make_plot(analysis, comparison)
    
    def make_plot(self, analysis, comparison):
        """
        Makes a plot.
        
        Parameters:
            analysis (str): Which analysis to run.
            comparison (str): Which comparison to run.
        """
        if len(analysis.get_projections()) == 1:
            comparison.data[0].mpl().data_hist(label="MicroBooNE data")
            comparison.predictions[0].mpl().hist(histtype="step",label=r"Achilles Prediction, $\chi^{2}=$"f"{comparison.likelihood():.3}")
            print(comparison.likelihood())
            plt.legend()
            plt.plot()
            plt.show()
        elif len(analysis.get_projections()) == 2:
            d2d = comparison.data[0]
            p2d = comparison.predictions[0]

            fig, axes = plt.subplots(2,2, figsize=(8,8))
            axes = axes.reshape(1,4).squeeze()

            proj_y_bins = d2d.project(1).binning.bins
            proj_y_bin_centers = np.array(pn.Binning.get_bin_centers1D(proj_y_bins))

            for i,y in enumerate(proj_y_bin_centers):
                d2d.slice(1,y).mpl().data_hist(label="MicroBooNE data",plot_axis=axes[i])
                p2d.slice(1,y).mpl().hist(histtype="step",label=r"Achilles Prediction, $\chi^{2}=$"f"{comparison.likelihood():.3}",plot_axis=axes[i])

            print(f"test statistic = {comparison.likelihood()}")
            axes[1].legend()
            fig.tight_layout()
            fig.show()
        else:
            raise ValueError("Higher than 2D distributions are not supported")
                        