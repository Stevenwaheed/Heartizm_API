# API packages
from flask import Flask, jsonify, request
import requests 
from flask_restful import Api, Resource, reqparse
import requests

# DataFrame packages
import numpy as np
import pandas as pd

# heart packages
import neurokit2 as nk
import heartpy

# ML packages
from sklearn.ensemble import ExtraTreesClassifier
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score

from sklearn.model_selection import train_test_split

# others
import os
import joblib
import json

class ECG:
    def __init__(self, original_signal):
        self.original_signal = original_signal
        
    
    def Distances(self, X1, Y1, X2, Y2):
        distances_results = []
        
        for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):
            result = np.math.sqrt((x2 - x1)**2 + (y2-y1)**2)
            distances_results.append(result)
            
        return distances_results


    def Slope(self, X1, Y1, X2, Y2):
        slope_results = []
        
        for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):

            result = (y2 - y1) / (x2 - x1)
            slope_results.append(result)
            
        return slope_results


    def Amplitudes(self, peak1, peak2):
        amplitudes = np.abs(peak1 - peak2)
        return amplitudes


    def intervals(self, Peaks1, Peaks2):
        res = np.abs(Peaks2 - Peaks1)
        return res


    def get_ECG_features(self, peaks1, peaks2):
        signal = self.edit_dataframe()
        signal.dropna(inplace=True)
        
        X1 = peaks2
        # print(peaks2.values.flatten())
        Y1 = signal.iloc[peaks2.values.flatten(), 0]

        X2 = peaks1
        Y2 = signal.iloc[peaks1.values.flatten(), 0]
        
        # Calculate Distances
        distances = self.Distances(X1, Y1, X2, Y2)
        
        # Calculate Slope
        slopes = self.Slope(X1, Y1, X2, Y2)

        return distances, slopes


    def get_sample_rate(self):
        sample_rate = int(self.original_signal.iloc[7, 1].split('.')[0])
        return sample_rate

    def get_person_name(self):
        person_name = self.original_signal.columns[1]
        return person_name

    def edit_dataframe(self):
        cols = self.original_signal.columns
        original_signal_1 = self.original_signal.drop(cols[1], axis=1)

        original_signal_2 = original_signal_1.drop(range(0, 10))
        original_signal_2['signals'] = original_signal_2['Name']
        
        original_signal_3 = original_signal_2.drop('Name', axis=1)
        original_signal_3['signals'] = original_signal_3['signals'].astype('float')
        
        
        return original_signal_3


    def remove_nulls(self, peaks, rpeaks):
        
        if len(peaks) > len(rpeaks):
            TF_selection = pd.DataFrame(peaks[:len(rpeaks)]).notna().values.flatten()
            new_peaks = pd.DataFrame(peaks[:len(rpeaks)]).dropna()
            new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
            
            return new_peaks.astype('int'), new_rpeaks.astype('int')
        
        elif len(peaks) < len(rpeaks):
            TF_selection = pd.DataFrame(peaks).notna().values.flatten()
            new_peaks = pd.DataFrame(peaks).dropna()
            rpeaks = rpeaks[:len(TF_selection)]
            new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
            
            return new_peaks.astype('int'), new_rpeaks.astype('int')
        
        else:
            total_df = pd.DataFrame(columns=['ECG_R_Peaks'])
            
            total_df['ECG_R_Peaks'] = rpeaks
            total_df['ECG_Peaks'] = peaks
            
            total_df.dropna(inplace=True)
            
            return total_df['ECG_Peaks'].astype('int'), total_df['ECG_R_Peaks'].astype('int')
        

    def main_features(self):
        sample_rate = self.get_sample_rate()
        signal = self.edit_dataframe()
        signals, _ = nk.ecg_process(signal.values.flatten(), sampling_rate=sample_rate)
        signals, _ = nk.ecg_process(signals['ECG_Clean'][2000:], sampling_rate=sample_rate)
        
        filtered_data = heartpy.filtering.filter_signal(signals.iloc[:, 1], filtertype='bandpass', cutoff=[2.5, 40], sample_rate=sample_rate, order=3)
        corrected_data = heartpy.hampel_correcter(filtered_data, sample_rate=sample_rate)
        final_signal = np.array(filtered_data)+np.array(corrected_data)
        
        filtered_data2 = heartpy.filtering.filter_signal(final_signal, filtertype='bandpass', cutoff=[3, 20], sample_rate=sample_rate, order=3)
        corrected_data2 = heartpy.filtering.hampel_correcter(filtered_data2, sample_rate=sample_rate)
        final_signal2 = np.array(filtered_data2) + np.array(corrected_data2)
        
        rr_peaks = nk.ecg_findpeaks(final_signal2, sampling_rate=sample_rate)
        
        _, features = nk.ecg_delineate(final_signal2, sampling_rate=sample_rate, method='peak')
        
        return signals, rr_peaks, features
        

    def feature_exctraction(self):
        Extracted_Features_DF = pd.DataFrame(columns=[
        'PR Distances', 'PR Slope', 'PR Amplitude',
        'PQ Distances', 'PQ Slope', 'PQ Amplitude',
        'QS Distances', 'QS Slope', 'QS Amplitude',
        'ST Distances', 'ST Slope', 'ST Amplitude',
        'RT Distances', 'RT Slope', 'RT Amplitude',

        'PS Amplitude', 'PT Amplitude', 'TQ Amplitude',
        'QR Amplitude', 'RS Amplitude'
        ])
        
        _, rr_peaks, features = self.main_features()
        
        
        p_peaks, pr_peaks = self.remove_nulls(features['ECG_P_Peaks'], rr_peaks['ECG_R_Peaks'])
        q_peaks, qr_peaks = self.remove_nulls(features['ECG_Q_Peaks'], rr_peaks['ECG_R_Peaks'])
        s_peaks, sr_peaks = self.remove_nulls(features['ECG_S_Peaks'], rr_peaks['ECG_R_Peaks'])
        t_peaks, tr_peaks = self.remove_nulls(features['ECG_T_Peaks'], rr_peaks['ECG_R_Peaks'])
 
        
        # Features between PR
        PR_distances = self.get_ECG_features(pr_peaks, p_peaks)[0]
        PR_slopes = self.get_ECG_features(pr_peaks, p_peaks)[1]
        PR_amplitudes = self.Amplitudes(pr_peaks.values.flatten(), p_peaks.values.flatten())

        PQ_distances = self.get_ECG_features(p_peaks, q_peaks)[0]
        PQ_slopes = self.get_ECG_features(p_peaks, q_peaks)[1]
        PQ_amplitudes = self.Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))

        QS_distances = self.get_ECG_features(q_peaks, s_peaks)[0]
        QS_slopes = self.get_ECG_features(q_peaks, s_peaks)[1]
        QS_amplitudes = self.Amplitudes(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))

        RT_distances = self.get_ECG_features(tr_peaks, t_peaks)[0]
        RT_slopes = self.get_ECG_features(tr_peaks, t_peaks)[1]
        RT_amplitudes = self.Amplitudes(tr_peaks.values.flatten(), t_peaks.values.flatten())

        ST_distances = self.get_ECG_features(s_peaks, t_peaks)[0]
        ST_slopes = self.get_ECG_features(s_peaks, t_peaks)[1]
        ST_amplitudes = self.Amplitudes(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))
    
        PS_amplitudes = self.Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
        PT_amplitudes = self.Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_P_Peaks']))
        TQ_amplitudes = self.Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_Q_Peaks']))
        RQ_amplitudes = self.Amplitudes(q_peaks.values.flatten(), qr_peaks.values.flatten())
        RS_amplitudes = self.Amplitudes(sr_peaks.values.flatten(), s_peaks.values.flatten())
    
        QR_interval = self.intervals(q_peaks.values.flatten(), qr_peaks.values.flatten())
        RS_interval = self.intervals(sr_peaks.values.flatten(), s_peaks.values.flatten())
        PQ_interval = self.intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))
        QS_interval = self.intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))
        PS_interval = self.intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
        PR_interval = self.intervals(p_peaks.values.flatten(), pr_peaks.values.flatten())
        ST_interval = self.intervals(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))
        QT_interval = self.intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_T_Peaks']))
        RT_interval = self.intervals(tr_peaks.values.flatten(), t_peaks.values.flatten())
        PT_interval = self.intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_T_Peaks']))
        
        lengths = [len(PR_distances), len(PR_slopes), len(PR_amplitudes)
           , len(PQ_distances), len(PQ_slopes), len(PQ_amplitudes)
           , len(QS_distances), len(QS_slopes), len(QS_amplitudes)
           , len(ST_distances), len(ST_slopes), len(ST_amplitudes)
           , len(RT_distances), len(RT_slopes), len(RT_amplitudes)
           , len(PS_amplitudes), len(PT_amplitudes), len(TQ_amplitudes)
           , len(RQ_amplitudes), len(RS_amplitudes)
           
           , len(QR_interval), len(RS_interval), len(PQ_interval)
           , len(QS_interval), len(PS_interval), len(PR_interval)
           , len(ST_interval), len(QT_interval), len(RT_interval)
           , len(PT_interval)
          ]

        minimum = min(lengths) - 1


        Extracted_Features_DF['PR Distances'] = PR_distances[:minimum]
        Extracted_Features_DF['PR Slope'] = PR_slopes[:minimum]
        Extracted_Features_DF['PR Amplitude'] = PR_amplitudes[:minimum]

        Extracted_Features_DF['PQ Distances'] = PQ_distances[:minimum]
        Extracted_Features_DF['PQ Slope'] = PQ_slopes[:minimum]
        Extracted_Features_DF['PQ Amplitude'] = PQ_amplitudes[:minimum]

        Extracted_Features_DF['QS Distances'] = QS_distances[:minimum] 
        Extracted_Features_DF['QS Slope'] = QS_slopes[:minimum]
        Extracted_Features_DF['QS Amplitude'] = QS_amplitudes[:minimum]

        Extracted_Features_DF['ST Distances'] = ST_distances[:minimum]
        Extracted_Features_DF['ST Slope'] = ST_slopes[:minimum]
        Extracted_Features_DF['ST Amplitude'] = ST_amplitudes[:minimum]

        Extracted_Features_DF['RT Distances'] = RT_distances[:minimum]
        Extracted_Features_DF['RT Slope'] = RT_slopes[:minimum]
        Extracted_Features_DF['RT Amplitude'] = RT_amplitudes[:minimum]

        Extracted_Features_DF['PS Amplitude'] = PS_amplitudes[:minimum]
        Extracted_Features_DF['PT Amplitude'] = PT_amplitudes[:minimum]
        Extracted_Features_DF['TQ Amplitude'] = TQ_amplitudes[:minimum]
        Extracted_Features_DF['QR Amplitude'] = RQ_amplitudes[:minimum]
        Extracted_Features_DF['RS Amplitude'] = RS_amplitudes[:minimum]


        Extracted_Features_DF['QR Interval'] = QR_interval[:minimum]
        Extracted_Features_DF['RS Interval'] = RS_interval[:minimum]
        Extracted_Features_DF['PQ Interval'] = PQ_interval[:minimum]
        Extracted_Features_DF['QS Interval'] = QS_interval[:minimum]
        Extracted_Features_DF['PS Interval'] = PS_interval[:minimum]
        Extracted_Features_DF['PR Interval'] = PR_interval[:minimum]
        Extracted_Features_DF['ST Interval'] = ST_interval[:minimum]
        Extracted_Features_DF['QT Interval'] = QT_interval[:minimum]
        Extracted_Features_DF['RT Interval'] = RT_interval[:minimum]
        Extracted_Features_DF['PT Interval'] = PT_interval[:minimum]
       
        return Extracted_Features_DF
    
    def labled_feature_exctraction(self):
        features = self.feature_exctraction()
        label = self.get_person_name()
        merged_df = pd.concat([features, pd.Series([label]*features.shape[0])], axis=1)
        
        return merged_df
        



model_path = 'C:\\Users\\Steven20367691\\Desktop\\new prototype 1\\'
# file_path = 'C:\\Users\\Steven20367691\\Desktop\\Team ECG\\Mira\\'
app = Flask(__name__)
api = Api(app)

extracted_features = pd.DataFrame()

# class ECG_API(Resource):
#     def post(self, file_name):
#         global extracted_features
#         original_ecg_signal = pd.read_csv(file_path+file_name)
        
#         ecg_heart = ECG(original_ecg_signal)
        
#         extracted_features = ecg_heart.feature_exctraction()
#         return ''

#     def get(self, file_name):
        
#         ExtraTree_model = joblib.load(model_path + 'Extra tree banha 2.h5')
#         extracted_features.dropna(inplace=True)
        
#         preds = ExtraTree_model.predict(extracted_features)
#         print(preds)
#         predictions = pd.DataFrame({'Results': preds})
#         return {'Results': predictions.value_counts().index[0][0]}


# api.add_resource(ECG_API, '/<string:file_name>')


def convert_json_dict(ecg_json):
    ecg_dict = {}
    for column in ecg_json.keys():
        ecg_dict[column] = list(ecg_json[column].values())
        
    ecg_df = pd.DataFrame(ecg_dict)
    return ecg_df


@app.route('/authentication', methods=['POST'])
def post():
    global extracted_features
    
    json_data = json.loads(request.data)
    ecg_df = convert_json_dict(json_data)
    ecg_heart = ECG(ecg_df)
    extracted_features = ecg_heart.feature_exctraction()
    
    return ' '


@app.route('/authentication', methods=['GET'])
def get():
    ExtraTree_model = joblib.load(model_path + 'Extra tree test 11 (97) very good.h5')
    extracted_features.dropna(inplace=True)

    preds = ExtraTree_model.predict(extracted_features)
    predictions = pd.DataFrame({'Results': preds})
    print(predictions)
    return jsonify({'Results': predictions.value_counts().index[0][0]})



all_features = pd.DataFrame()

@app.route('/authentication/store', methods=['POST'])
def store():
    global extracted_features
    global all_features
    
    json_data = json.loads(request.data)
    ecg_df = convert_json_dict(json_data)
    ecg_heart = ECG(ecg_df)
    extracted_features = ecg_heart.labled_feature_exctraction()
    
    if len(all_features) == 0:
        all_features = extracted_features
    else:
        all_features = pd.concat([all_features, extracted_features])
        
    print(all_features)
    
    return ' '


@app.route('/authentication/train', methods=['GET'])
def train():
    global all_features

    df = all_features.dropna()
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ExtraTree = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=2)
    ExtraTree.fit(X_train, y_train)
    # preds = ExtraTree.predict(X_test)

    # ExtraTree model
    ExtraTree_preds = ExtraTree.predict(X_test)
    print('accuracy_score:', accuracy_score(ExtraTree_preds, y_test.values))
    print('f1_score:', f1_score(y_test.values, ExtraTree_preds, average='weighted'))
    print('recall_score:', recall_score(ExtraTree_preds, y_test.values, average='weighted'))
    print('precision_score:', precision_score(ExtraTree_preds, y_test.values, average='weighted'))
    
    return jsonify({'Performance': f'- Accuracy Score: {accuracy_score(ExtraTree_preds, y_test.values)}'+
                    f'\n- F1 Score: {f1_score(ExtraTree_preds, y_test.values, average="weighted")}'
                    +f'\n- Recall Score: {recall_score(ExtraTree_preds, y_test.values, average="weighted")}'
                    +f'\n- Precision Score: {precision_score(ExtraTree_preds, y_test.values, average="weighted")}'})




if __name__ == '__main__':
    app.run(debug=True)
