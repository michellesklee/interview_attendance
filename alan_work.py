import numpy as np
import pandas as pd

df = pd.read_csv('Interview.csv')
cols = df.columns
cols = [col.lower().replace(" ", "_") for col in cols]
df.columns = cols
df.drop(['date_of_interview', 'name(cand_id)', 'nature_of_skillset', 'unnamed:_23', 'unnamed:_24', 'unnamed:_25', 'unnamed:_26', 'unnamed:_27'], axis=1, inplace=True)
df = df.fillna(0)
df.rename(index=str, columns={"have_you_obtained_the_necessary_permission_to_start_at_the_required_time": "permission_start_on_time", "hope_there_will_be_no_unscheduled_meetings": "hope_no_unsch_meets", "can_i_call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview": "call_three_hrs_before", "can_i_have_an_alternative_number/_desk_number._i_assure_you_that_i_will_not_trouble_you_too_much": "alt_num_given", "have_you_taken_a_printout_of_your_updated_resume._have_you_read_the_jd_and_understood_the_same": "come_prepared", "are_you_clear_with_the_venue_details_and_the_landmark.": "can_find_interview_loc", "has_the_call_letter_been_shared": "call_letter_shared"}, inplace=True)

yes_no_cols = ['permission_start_on_time', 'hope_no_unsch_meets', 'call_three_hrs_before', 'alt_num_given', 'come_prepared', 'can_find_interview_loc', 'call_letter_shared', 'expected_attendance', 'observed_attendance']


for col in yes_no_cols:
    df[col] = df[col] == 'Yes'
df.drop(df.index[1233], inplace=True)

industry_map = {
    'IT Services':'IT',
    'IT Products and Services':'IT',
    'IT': 'IT',
    'Pharmaceuticals': 'Pharmaceuticals',
    'BFSI': 'BFSI',
    'Electronics': 'Electronics',
    'Telecom': 'Telecom'
    }

df['industry'] = df['industry'].map(lambda x: industry_map[x])

location_map = {
    'Chennai':'Chennai',
    'Gurgaon':'Gurgaon',
    'Bangalore': 'Bangalore',
    'Hyderabad': 'Hyderabad',
    'Gurgaonr': 'Gurgaon',
    'Delhi': 'Delhi',
    'chennai': 'Chennai',
    '- Cochin- ': 'Cochin',
    'Noida': 'Noida',
    'CHENNAI': 'Chennai',
    'chennai ': 'Chennai'
    }
df['location'] = df['location'].map(lambda x: location_map[x])

interview_type_map = {
    'Scheduled Walkin':'Scheduled_Walkin',
    'Scheduled ':'Scheduled',
    'Walkin': 'Walkin',
    'Scheduled Walk In': 'Scheduled_Walkin',
    'Sceduled walkin': 'Scheduled_Walkin',
    'Walkin ': 'Walkin'
    }
df['interview_type'] = df['interview_type'].map(lambda x: interview_type_map[x])

candidate_curr_location_map = {
    'Chennai':'Chennai',
    'Gurgaon':'Gurgaon',
    'Bangalore': 'Bangalore',
    'Hyderabad': 'Hyderabad',
    'Delhi': 'Delhi',
    'chennai': 'Chennai',
    '- Cochin- ': 'Cochin',
    'Noida': 'Noida',
    'CHENNAI': 'Chennai',
    'chennai ': 'Chennai'
    }
df['candidate_current_location'] = df['candidate_current_location'].map(lambda x: candidate_curr_location_map[x])

candidate_job_location_map = {
    'Chennai':'Chennai',
    'Gurgaon':'Gurgaon',
    'Bangalore': 'Bangalore',
    'Hosur': 'Hosur',
    'Visakapatinam': 'Visakapatinam',
    '- Cochin- ': 'Cochin',
    'Noida': 'Noida',
    }
df['candidate_job_location'] = df['candidate_job_location'].map(lambda x: candidate_job_location_map[x])

interview_venue_map = {
    'Hosur':'Hosur',
    'Gurgaon':'Gurgaon',
    'Bangalore': 'Bangalore',
    'Hyderabad': 'Hyderabad',
    'Chennai': 'Chennai',
    '- Cochin- ': 'Cochin',
    'Noida': 'Noida',
    }
df['interview_venue'] = df['interview_venue'].map(lambda x: interview_venue_map[x])

candidate_native_location_map = {
    'Hosur': 'Hosur',
    'Chennai': 'Chennai',
    'Gurgaon': 'Gurgaon',
    'Noida': 'Noida',
    'Delhi /NCR': 'Delhi',
    'Trivandrum': 'Trivandrum',
    'Cochin': 'Cochin',
    'Bangalore': 'Bangalore',
    'Coimbatore': 'Coimbatore',
    'Salem': 'Salem',
    'Tanjore': 'Tanjore',
    'Hyderabad': 'Hyderabad',
    'Mumbai': 'Mumbai',
    'Pune': 'Pune',
    'Kolkata': 'Kolkata',
    'Allahabad': 'Allahabad',
    'Cuttack': 'Cuttack',
    'Visakapatinam': 'Visakapatinam',
    'Belgaum': 'Belgaum',
    'Patna': 'Patna',
    'Chitoor': 'Chitoor',
    'Anantapur': 'Anantapur',
    'Warangal': 'Warangal',
    'Ahmedabad': 'Ahmedabad',
    'Kurnool': 'Kurnool',
    'Vijayawada': 'Vijayawada',
    'Vellore': 'TricVellorehy',
    'Pondicherry': 'Pondicherry',
    'Nagercoil': 'Nagercoil',
    'Agra': 'Agra',
    'Bhubaneshwar': 'Bhubaneshwar',
    'Ghaziabad': 'Ghaziabad',
    'Baddi': 'Baddi',
    'Tuticorin': 'Tuticorin',
    'Tirupati': 'Tirupati',
    'Faizabad': 'Faizabad',
    'Ambur': 'Ambur',
    'Chandigarh': 'Chandigarh',
    'Mysore': 'Mysore',
    'Hissar': 'Hissar',
    'Delhi': 'Delhi',
    'Kanpur': 'Kanpur',
    'Lucknow': 'Lucknow',
    '- Cochin- ': 'Cochin',
    'Trichy': 'Trichy',
    'Panjim': 'Panjim'
    }
df['candidate_native_location'] = df['candidate_native_location'].map(lambda x: candidate_native_location_map[x])


#df = pd.read_pickle('/Users/alanteran/galvanize/interview_attendance/pickled.pkl')
