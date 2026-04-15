import pandas as pd
from pathlib import Path


def get_revision_text(base_dir, file_id):
    match = list(Path(base_dir).glob(f"part*/{file_id}.txt"))

    if not match:
        return None

    return match[0].read_text(encoding="utf-8", errors="ignore")


PATH_TO_2011_VANDALISM_CORPUS = 'Yout path to the 2011 vadalism corpus here.'
PATH_TO_ARTICLE_REVISIONS_ENGLISH = 'Your path for english revision here.'
PATH_TO_ARTICLE_REVISIONS_GERMAN = 'Your path for german revision here.'
PATH_TO_ARTICLE_REVISIONS_SPANISH = 'Your path for spanish revision here.'
PATH_TO_2010_DATA = 'Your path to the 2010 vandalism dataset'

OUTPUT_PATH = 'PATH TO THE PROCESSED OUTPUT .parquet'

if __name__ == '__main__':

    # English edits.
    edits_df_en = pd.read_csv(PATH_TO_2011_VANDALISM_CORPUS + '/edits-en.csv')
    parsed_en = {
        'edit_id': [],
        'editor_id': [],
        'timestamp': [],
        'article_id': [],
        'edit_size': [],
        'language': [],
        'is_vandalism': [],
    }
    for _, row in edits_df_en.iterrows():
        parsed_en['edit_id'].append(row['editid'])
        parsed_en['editor_id'].append(row['editor'])
        parsed_en['timestamp'].append(pd.to_datetime(row['edittime']))
        parsed_en['article_id'].append(row['articleid'])

        old_text_len = len(
            get_revision_text(PATH_TO_ARTICLE_REVISIONS_ENGLISH,
                              row['oldrevisionid']))
        new_text_len = len(
            get_revision_text(PATH_TO_ARTICLE_REVISIONS_ENGLISH,
                              row['newrevisionid']))
        parsed_en['edit_size'].append(new_text_len - old_text_len)
        parsed_en['is_vandalism'].append(int(row['class'] != 'regular'))
        parsed_en['language'].append('en')

    edits_df_de = pd.read_csv(PATH_TO_2011_VANDALISM_CORPUS + '/edits-de.csv')
    parsed_de = {
        'edit_id': [],
        'editor_id': [],
        'timestamp': [],
        'article_id': [],
        'edit_size': [],
        'language': [],
        'is_vandalism': [],
    }
    for _, row in edits_df_de.iterrows():
        parsed_de['edit_id'].append(row['editid'])
        parsed_de['editor_id'].append(row['editor'])
        parsed_de['timestamp'].append(pd.to_datetime(row['edittime']))
        parsed_de['article_id'].append(row['articleid'])

        old_text_len = len(
            get_revision_text(PATH_TO_ARTICLE_REVISIONS_GERMAN,
                              row['oldrevisionid']))
        new_text_len = len(
            get_revision_text(PATH_TO_ARTICLE_REVISIONS_GERMAN,
                              row['newrevisionid']))
        parsed_de['edit_size'].append(new_text_len - old_text_len)
        parsed_de['is_vandalism'].append(int(row['class'] != 'regular'))
        parsed_de['language'].append('de')

    edits_df_es = pd.read_csv(PATH_TO_2011_VANDALISM_CORPUS + '/edits-es.csv')
    parsed_es = {
        'edit_id': [],
        'editor_id': [],
        'timestamp': [],
        'article_id': [],
        'edit_size': [],
        'language': [],
        'is_vandalism': [],
    }
    for _, row in edits_df_es.iterrows():
        parsed_es['edit_id'].append(row['editid'])
        parsed_es['editor_id'].append(row['editor'])
        parsed_es['timestamp'].append(pd.to_datetime(row['edittime']))
        parsed_es['article_id'].append(row['articleid'])

        old_text_len = len(
            get_revision_text(PATH_TO_ARTICLE_REVISIONS_SPANISH,
                              row['oldrevisionid']))
        new_text_len = len(
            get_revision_text(PATH_TO_ARTICLE_REVISIONS_SPANISH,
                              row['newrevisionid']))
        parsed_es['edit_size'].append(new_text_len - old_text_len)
        parsed_es['is_vandalism'].append(int(row['class'] != 'regular'))
        parsed_es['language'].append('es')

    all_languages_parsed = pd.concat([
        pd.DataFrame(parsed_en),
        pd.DataFrame(parsed_es),
        pd.DataFrame(parsed_de),
    ])

    # 2010 parsing
    edits_df = pd.read_csv(PATH_TO_2010_DATA + '/edits.csv')
    gold_anotations_df = pd.read_csv(PATH_TO_2010_DATA +
                                     '/gold-annotations.csv')
    merged_df = pd.merge(edits_df,
                         gold_anotations_df,
                         on='editid',
                         how='inner')
    parsed = {
        'edit_id': [],
        'editor_id': [],
        'timestamp': [],
        'article_id': [],
        'edit_size': [],
        'is_vandalism': [],
        'language': []
    }
    for _, row in merged_df.iterrows():
        parsed['edit_id'].append(row['editid'])
        parsed['editor_id'].append(row['editor'])
        parsed['timestamp'].append(pd.to_datetime(row['edittime']))
        parsed['article_id'].append(row['articleid'])

        old_text_len = len(
            get_revision_text(PATH_TO_2010_DATA + '/article-revisions',
                              row['oldrevisionid']))
        new_text_len = len(
            get_revision_text(PATH_TO_2010_DATA + '/article-revisions',
                              row['newrevisionid']))
        parsed['edit_size'].append(new_text_len - old_text_len)
        parsed['is_vandalism'].append(int(row['class'] != 'regular'))
        parsed['language'].append('en')

    # Merging all data
    all_data = pd.concat([all_languages_parsed, pd.DataFrame(parsed)])
    all_data.to_parquet(OUTPUT_PATH)
