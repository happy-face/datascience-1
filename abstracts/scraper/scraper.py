import arxivscraper.arxivscraper as ax
import pandas as pd


topics = ['cs','econ','eess','math','physics','q-bio','q-fin','stat']

def scrape_arxiv(topic):
    scraper = ax.Scraper(category=topic, date_from='2018-02-27', date_until='2018-03-27')
    output = scraper.scrape()
    #print(output)
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(output, columns=cols)
    return df

'''df_cs = scrape_arxiv(topics[0])
df_econ = scrape_arxiv(topics[1])
df_eess = scrape_arxiv(topics[2])
df_math = scrape_arxiv(topics[3])
df_physics = scrape_arxiv(topics[4])
df_qbio = scrape_arxiv(topics[5])
df_qfin = scrape_arxiv(topics[6])
df_stat = scrape_arxiv(topics[7])'''

#all_df = [df_cs, df_econ, df_eess, df_math, df_physics, df_qbio, df_qfin, df_stat]
#result = pd.concat(all_df)

frames = [scrape_arxiv(topic) for topic in topics]
result = pd.concat(frames)

#result.to_csv('C:\insight_program\\abstracts\TrainingData.csv')