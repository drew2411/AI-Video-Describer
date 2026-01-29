# AI-Video-Describer

Setup:
python -m venv venv
venv/Scripts/Activate.ps1
pip install -r requirements.txt
python analyse.py

All other folders  hold logs and older code

//
Main Code is in core/
Main file is mad_ad_genereator_final.py

the files named chroma are to : 
store the movies into the vector db (chroma_index)
resume indexing  if only partial indexing was done (chroma_resume_frames)
see the contents of chroma (test_chroma)

EVALUATION:
evaluate_ads.py - 
evaluated ads basedo n BLEU and ROUGE, 
which we should change to cosine similarity OR manual verification
BLEU and ROUGE are used to find exact words in common between two sentences 
cosine similarity checks for meaning


HELPER SCRIPTS
find_good_movies is to find segments in  movies that have a certain number of ADs
read the main() function to get a better idea of the parameters
