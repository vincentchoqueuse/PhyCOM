import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import pandas as pd
import time
from csv import writer


def plot_constellation(data):
    output = np.ravel(np.array(data))
    figure = plt.figure()
    plt.plot(np.real(output), np.imag(output), "x")
    plt.axis('equal')
    return figure


class MC_Report_Writer():

    def __init__(self, folder, filename, nb_trials, parameter_name="params", summary=None):
        self.summary = summary
        self.filename = "{}/{}".format(folder,filename)
        self.parameter_name = parameter_name
        self.mc = []
        self.nb_trials = nb_trials
        self.stop_time = None
        self.json_data = None

    def start(self):
        now = datetime.now()
        self.status = "In Progress"
        self.start_time = now.strftime("%H:%M:%S")

    def stop(self):
        now = datetime.now()
        self.status = "Finished"
        self.stop_time = now.strftime("%H:%M:%S")
        self.export_json()

    def start_mc(self, index):
        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        self.mc_start_time = now
        self.current_report = {"value": index, "current_trial": 0, "status": "In Progress", "start_time": start_time}

    def update_current_mc(self, current_trial):
        self.current_report["current_trial"] = current_trial
        mc = self.mc + [self.current_report]
        self.export_json(mc)

    def stop_current_mc(self):
        now = datetime.now()
        diff = now - self.mc_start_time

        stop_time = now.strftime("%H:%M:%S")
        elapsed_time = str(diff)
        self.current_report["stop_time"] = stop_time
        self.current_report["elapsed_time"] = elapsed_time
        self.current_report["current_trial"] = self.nb_trials
        self.current_report["status"] = "Finished"
        self.mc.append(self.current_report)
        self.export_json()

    def export_json(self, mc=None):
        if mc is None:
            mc = self.mc

        json_data = {}
        json_data["summary"] = self.summary
        json_data["status"] = self.status
        json_data["start_time"] = self.start_time
        json_data["stop_time"] = self.stop_time
        json_data["nb_trials"] = self.nb_trials
        json_data["parameter"] = self.parameter_name
        json_data["mc"] = mc

        self.json_data = json_data
        self.save()

    def save(self):
        with open(self.filename, 'w') as outfile:
            json.dump(self.json_data, outfile)


class Data_Writer():

    def __init__(self, folder, filename, index, columns):
        self.filename = "{}/{}".format(folder,filename)
        self.index = index
        self.columns = columns
        self.data = []

    def add(self, row):
        self.data.append(row.tolist())
        status = self.save()
        return status

    def save(self):
        index = self.index[:len(self.data)]
        status = pd.DataFrame(self.data, index=index, columns=self.columns).to_csv(self.filename)
        return status

