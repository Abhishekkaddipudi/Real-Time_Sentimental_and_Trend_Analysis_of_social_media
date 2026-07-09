FROM python:3.10
COPY . /dashboard
WORKDIR /dashboard
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD streamlit run sqlDB_dashboard.py --server.fileWatcherType none