import pandas as pd
import pytest

@pytest.fixture
def df_min():
    # first two rows are exact duplicates for dedup tests
    return pd.DataFrame({
        "PatientID":[1,1,2,3], "DoctorInCharge":["X","X","Y","Z"],
        "Age":[20,20,30,40], "BMI":[22.0,22.0,30.0,28.0],
        "PhysicalActivity":[0.3,0.3,0.2,0.9], "DietQuality":[6.0,6.0,4.0,7.0],
        "SleepQuality":[8.0,8.0,5.0,7.0], "PollutionExposure":[0.4,0.4,0.3,0.9],
        "PollenExposure":[0.5,0.5,0.2,0.8], "DustExposure":[0.4,0.4,0.3,0.7],
        "LungFunctionFEV1":[3.0,3.0,2.5,3.5], "LungFunctionFVC":[3.2,3.2,2.6,3.6],
        "Gender":[1,1,0,1], "Ethnicity":[1,1,2,3], "EducationLevel":[1,1,2,3],
        "Smoking":[0,0,1,1], "PetAllergy":[1,1,0,1], "FamilyHistoryAsthma":[1,1,0,1],
        "HistoryOfAllergies":[0,0,1,0], "Eczema":[1,1,0,1], "HayFever":[0,0,1,0],
        "GastroesophagealReflux":[1,1,0,1], "Wheezing":[1,1,0,1],
        "ShortnessOfBreath":[1,1,1,0], "ChestTightness":[1,1,0,1],
        "Coughing":[1,1,0,1], "NighttimeSymptoms":[0,0,1,1], "ExerciseInduced":[1,1,0,1],
        "Diagnosis":[1,1,0,0], "CONST":[5,5,5,5], "AgeCopy":[20,20,30,40],
    })
