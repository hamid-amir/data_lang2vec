import json


method = 'rf'
save_dir = 'result'

target_features = [
    'S_DEMONSTRATIVE_WORD_BEFORE_NOUN', 'S_INDEFINITE_AFFIX', 'S_SEX_MARK', 'S_VOS', 'S_OBJECT_BEFORE_VERB',
    'S_OBJECT_AFTER_VERB', 'S_NEGATIVE_WORD_AFTER_OBJECT', 'S_NUMCLASS_MARK', 'P_UVULAR_STOPS', 'P_CLICKS',
    'S_CASE_SUFFIX', 'S_XVO', 'S_NEGATIVE_WORD_BEFORE_SUBJECT', 'P_UVULAR_CONTINUANTS', 'S_POLARQ_AFFIX',
    'S_CASE_PROCLITIC', 'P_IMPLOSIVES', 'S_POLARQ_WORD', 'S_OBLIQUE_AFTER_VERB', 'S_DEFINITE_AFFIX',
    'S_INDEFINITE_WORD', 'P_VELAR_NASAL', 'P_NASAL_VOWELS', 'S_RELATIVE_AROUND_NOUN', 'P_BILABIALS',
    'S_POSSESSIVE_HEADMARK', 'S_SUBJECT_BEFORE_OBJECT', 'S_TEND_SUFFIX', 'P_TONE', 'S_OSV', 'P_VELAR_NASAL_INITIAL',
    'S_GENDER_MARK', 'P_FRICATIVES', 'P_VOICED_PLOSIVES', 'S_DEMONSTRATIVE_PREFIX', 'S_SUBORDINATOR_WORD_BEFORE_CLAUSE',
    'P_COMPLEX_CODAS', 'S_CASE_PREFIX', 'P_EJECTIVES', 'P_CODAS', 'S_OBLIQUE_BEFORE_VERB', 'S_OBJECT_HEADMARK',
    'S_ADJECTIVE_AFTER_NOUN', 'S_DEMONSTRATIVE_SUFFIX', 'S_NEGATIVE_PREFIX', 'S_CASE_MARK', 'S_COMITATIVE_VS_INSTRUMENTAL_MARK',
    'S_NEGATIVE_WORD_AFTER_VERB', 'S_SOV', 'S_NEGATIVE_WORD_ADJACENT_AFTER_VERB', 'S_NUMERAL_AFTER_NOUN', 'S_ADJECTIVE_WITHOUT_NOUN',
    'S_PLURAL_PREFIX', 'S_DEGREE_WORD_AFTER_ADJECTIVE', 'S_POSSESSIVE_DEPMARK', 'S_ADPOSITION_AFTER_NOUN', 'S_SUBJECT_BEFORE_VERB',
    'S_PLURAL_WORD', 'P_NASALS', 'S_TEND_PREFIX', 'S_SUBORDINATOR_WORD_AFTER_CLAUSE', 'S_OVS', 'S_NOMINATIVE_VS_ACCUSATIVE_MARK',
    'S_DEGREE_WORD_BEFORE_ADJECTIVE', 'S_RELATIVE_AFTER_NOUN', 'S_SVO', 'S_OXV', 'P_LATERAL_L', 'P_LABIAL_VELARS', 'S_DEMONSTRATIVE_WORD_AFTER_NOUN',
    'P_LATERAL_OBSTRUENTS', 'S_VSO', 'S_NEGATIVE_WORD_ADJACENT_BEFORE_VERB', 'S_POSSESSOR_AFTER_NOUN', 'P_PHARYNGEALS', 'P_VOICED_FRICATIVES',
    'S_OBLIQUE_BEFORE_OBJECT', 'S_POLARQ_MARK_FINAL', 'S_PLURAL_SUFFIX', 'S_NEGATIVE_WORD_AFTER_SUBJECT', 'S_SUBJECT_AFTER_VERB',
    'S_PERFECTIVE_VS_IMPERFECTIVE_MARK', 'S_NEGATIVE_WORD_FINAL', 'S_TAM_PREFIX', 'S_VOX', 'S_POSSESSIVE_PREFIX', 'P_GLOTTALIZED_RESONANTS',
    'S_PROSUBJECT_CLITIC', 'P_TH', 'S_NEGATIVE_SUFFIX', 'S_DEFINITE_WORD', 'S_FUTURE_AFFIX', 'S_RELATIVE_BEFORE_NOUN', 'S_XOV',
    'S_ERGATIVE_VS_ABSOLUTIVE_MARK', 'S_NEGATIVE_AFFIX', 'P_UVULARS', 'S_NUMERAL_BEFORE_NOUN', 'S_NEGATIVE_WORD_BEFORE_VERB', 'S_OVX',
    'S_CASE_ENCLITIC', 'S_POLARQ_MARK_SECOND', 'S_ADJECTIVE_BEFORE_NOUN', 'S_PROSUBJECT_WORD', 'S_ADPOSITION_BEFORE_NOUN',
    'S_POLARQ_MARK_INITIAL', 'S_NEGATIVE_WORD_INITIAL', 'S_TEND_DEPMARK', 'P_COMPLEX_ONSETS', 'S_POSSESSOR_BEFORE_NOUN',
    'P_FRONT_ROUND_VOWELS', 'P_LATERALS', 'S_PAST_VS_PRESENT_MARK', 'S_SUBJECT_AFTER_OBJECT', 'S_TEND_HEADMARK', 'S_OBJECT_DEPMARK',
    'S_NEGATIVE_WORD', 'S_TAM_SUFFIX', 'S_NEGATIVE_WORD_BEFORE_OBJECT', 'S_ANY_REDUP', 'S_PROSUBJECT_AFFIX', 'S_OBLIQUE_AFTER_OBJECT',
    'S_POSSESSIVE_SUFFIX', 'P_VOICE', 'S_SUBORDINATOR_SUFFIX',
]
c = 1
for target_feature in target_features:
    with open(f'{save_dir}/optuna_study_results_method_{method}_target_feature_{target_feature}.json') as f:
        d = json.load(f)
        d_with_miltale_data = [i for i in d if i['params']['miltale_data']]
        d_without_miltale_data = [i for i in d if not i['params']['miltale_data']]
        assert len(d_with_miltale_data) + len(d_without_miltale_data) == len(d)
        best_trial = max(d, key=lambda x:x['value'])
        if best_trial['params']['miltale_data']:
            best_trial_with_miltale_data = max(d_with_miltale_data, key=lambda x:x['value'])
            best_trial_without_miltale_data = max(d_without_miltale_data, key=lambda x:x['value'])

            assert best_trial_with_miltale_data['value'] == best_trial['value']
            assert best_trial_without_miltale_data['value'] <= best_trial['value']
            
            print(f"{c}: {target_feature} - {best_trial_with_miltale_data['value']} - {best_trial_without_miltale_data['value']}")
            c += 1