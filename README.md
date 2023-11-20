# human-benchmark
Automatic human evaluation for systems

## HOW TO RUN



```bash
docker-compose up -d
uvicorn main:app --port 8088

```
* Admin app
```bash
python -m streamlit run  /home/ubuntu/repos/Efficient-MT/web_app/admin/admin_app.py
```

* Rater app
```bash
python -m streamlit run  /home/ubuntu/repos/Efficient-MT/web_app/annotator/rater_app.py
```
