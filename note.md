.venv\Scripts\activate
pip freeze > requirements.txt   next

uvicorn app:app --reload    //local web server
python attack.py  //test for adverserial attack

http://127.0.0.1:8000/docs to upload and test the files/images