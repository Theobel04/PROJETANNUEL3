from icrawler.builtin import BingImageCrawler  
import os

merveilles = {
    "great_wall":       "Great Wall of China monument",
    "taj_mahal":        "Taj Mahal monument India",
    "christ_redeemer":  "Christ the Redeemer statue Brazil",
}

for folder, query in merveilles.items():
    print(f"Téléchargement : {query}")
    crawler = BingImageCrawler (storage={"root_dir": f"../dataset/{folder}"})
    crawler.crawl(keyword=query, max_num=150)

print("Dataset téléchargé !")