from tqdm import tqdm

# Globally defined feature to name maps
fn_maps = {
    "BloodUreaNitrogen": "Blood Urea Nitrogen",
    "UrineOutput": "Urine Output",
    "KReplacement": "Potassium Replacement",
    "DiastolicBloodPressure": "Diastolic Blood Pressure",
    "MeanBloodPressure": "Mean Blood Pressure",
    "MgReplacement": "Magnesium Replacement",
    "RespiratoryRate": "Respiratory Rate",
    "SystolicBloodPressure": "Systolic Blood Pressure",
    "AirwayPressure": "Airway Pressure",
    "CaReplacement": "Calcium Replacement",
    "MinuteVentilation": "Minute Ventilation",
    "PReplacement": "Phosphorus Replacement",
    "TidalVolume": "Tidal Volume",
    "CTScan": "CT Scan",
    "EnteralNutrition": "Enteral Nutrition",
    "PronePosition": "Prone Position",
    "UltraSound": "Ultra Sound",
    "CreatinineKinase": "Creatinine Kinase",
    "ICPMonitor": "Intracranial Pressure Monitor",
    "Transfusions": "Blood Transfusions",
    "HeartRate": "Heart Rate",
}


def merge_vaso(spm):
    for stay in spm:
        vaso_features = []
        vaso_tv = []
        for feature, tv in stay.items():
            if "Vaso" not in feature:
                continue
            else:
                vaso_tv.extend(tv)
                vaso_features.append(feature)
                
        for f in vaso_features:
            stay.pop(f)
        stay["Vasopressors"] = sorted(vaso_tv)
    
    return spm


def text_gen(past_hours, spm, mode, all_fns, binaryfns):
    spm = merge_vaso(spm)

    feature_text = []
    time_text = []

    collected_tv_full = []
    collected_fv_full = []

    for stay in tqdm(spm):
        collected_tv_org = {}
        collected_tv = {}
        collected_fv = {}

        for fn, values in stay.items():
            feature = fn.split("_")[0]
            if feature not in collected_tv:
                collected_tv[feature] = []
            
            if feature not in collected_tv_org:
                collected_tv_org[feature] = []
            
            for t, v in values:
                rounded_t = round(t)

                assert rounded_t <= 48
                if rounded_t not in collected_fv:
                    collected_fv[rounded_t] = []

                nv = v

                collected_tv_org[feature].append((t, v))

                collected_tv[feature].append((rounded_t, nv))
                collected_fv[rounded_t].append((feature, nv))
            
            collected_tv[feature] = sorted(collected_tv[feature])
        
        collected_tv_full.append(collected_tv)
        collected_fv_full.append(collected_fv)

        if mode == "feature":
            texts = []
            for feature in all_fns:
                p_feature = fn_maps.get(feature, feature)

                if feature not in collected_tv:
                    texts.append(f"{p_feature} has no values")
                    continue
                
                tv = collected_tv[feature]
                if len(tv) == 0:
                    texts.append(f"{p_feature} has no values")
                    continue
                
                prev_t = -1
                agg_v = []
                
                if feature in binaryfns or feature == "Vasopressors":
                    ind_t = f"Patient receives {p_feature} "
                else:
                    ind_t = f"{p_feature} has "

                for t, v in tv:
                    if t != prev_t:
                        if not agg_v:
                            prev_t = t
                            agg_v.append(v)
                            continue

                        if feature not in binaryfns and feature != "Vasopressors":
                            avg_v = sum(agg_v) / len(agg_v)
                            if avg_v % 1 == 0:
                                avg_v = int(avg_v)
                            else:
                                avg_v = round(avg_v, 3)

                            ind_t += f"value {avg_v} in hour {prev_t}, "
                        else:
                            ind_t += f"at hour {prev_t}, "
                        
                        agg_v = []
                        prev_t = t
                        agg_v.append(v)
                    else:
                        agg_v.append(v)

                if agg_v:
                    if feature not in binaryfns and feature != "Vasopressors":
                        avg_v = sum(agg_v) / len(agg_v)
                        if avg_v % 1 == 0:
                            avg_v = int(avg_v)
                        else:
                            avg_v = round(avg_v, 3)

                        ind_t += f"value {avg_v} in hour {prev_t}, "
                    else:
                        ind_t += f"at hour {prev_t}, "

                ind_t = ind_t[:-2]
                texts.append(ind_t)

            feature_text.append(texts)
        
        if mode == "time":
            texts = []
            # for i in range(1, min(past_hours, int(pt)) + 1):
            for i in range(1, past_hours + 1):
                if i not in collected_fv:
                    texts.append(f"No value in this hour")
                    continue
                
                fv = collected_fv[i]
                if len(fv) == 0:
                    texts.append(f"No value in this hour")
                    continue
                
                ind_f = ""
                ind_binary = ""

                prev_f = -1
                agg_v = []
                for f, v in fv:
                    if v % 1 == 0:
                        v = int(v)

                    if f != prev_f:
                        if not agg_v:
                            prev_f = f
                            agg_v.append(v)
                            continue

                        if prev_f in binaryfns or prev_f == "Vasopressors":
                            if prev_f in fn_maps:
                                ind_binary += f"Patient receives {fn_maps[prev_f]}, "
                            else:
                                ind_binary += f"Patient receives {prev_f}, "
                        else:
                            avg_v = sum(agg_v) / len(agg_v)

                            if avg_v % 1 == 0:
                                avg_v = int(avg_v)
                            else:
                                avg_v = round(avg_v, 3)

                            if prev_f in fn_maps:
                                ind_f += f"{fn_maps[prev_f]} has value {avg_v}, "
                            else:
                                ind_f += f"{prev_f} has value {avg_v}, "

                        agg_v = []
                        prev_f = f
                        agg_v.append(v)

                    else:
                        agg_v.append(v)
            
                if agg_v:
                    if prev_f in binaryfns or prev_f == "Vasopressors":
                        if prev_f in fn_maps:
                            ind_binary += f"Patient receives {fn_maps[prev_f]}, "
                        else:
                            ind_binary += f"Patient receives {prev_f}, "
                    else:
                        avg_v = sum(agg_v) / len(agg_v)
                        if avg_v % 1 == 0:
                            avg_v = int(avg_v)
                        else:
                            avg_v = round(avg_v, 3)

                        if prev_f in fn_maps:
                            ind_f += f"{fn_maps[prev_f]} has value {avg_v}, "
                        else:
                            ind_f += f"{prev_f} has value {avg_v}, "

                if not ind_f:
                    ind_binary = ind_binary[:-2]

                ind_f = ind_f[:-2]
                ind_f = ind_binary + ind_f

                texts.append(ind_f)
            
            time_text.append(texts)

    if mode == "feature":
        return feature_text

    if mode == "time":
        return time_text

    raise NotImplementedError(f"{mode} undefined")

import pickle
# mimic_path = "/voyager/datasets/physionet.org/files/mimiciv/3.0/proced/mimic_ml/mimic_spmx.pkl"
# mimic = pickle.load(open(mimic_path, "rb"))

hirid_path = "/voyager/datasets/ehr_datasets/hirid/hirid_ml/hirid_spmx.pkl"
hirid = pickle.load(open(hirid_path, "rb"))

all_fns = hirid["fns_dict"]["fn_order"]
binaryfns = hirid["fns_dict"]["binaryfns"]

results = text_gen(48, hirid["all_spmx"], "feature", all_fns, binaryfns)

all_sbf_texts = []
for texts in results:
    ind = ""

    for t in texts:
        if not t.endswith("has no values"):
            ind += f"{t}. "
    
    # if not all_sbf_texts:
    #     breakpoint()

    all_sbf_texts.append(ind)

pickle.dump(all_sbf_texts, open("/voyager/datasets/physionet.org/data_gen/hirid_sbf_short.pkl", "wb"))
    