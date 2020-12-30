FROM python:3.8.5-slim

# remember to expose the port your app'll be exposed on.
EXPOSE 8501
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . .

# run it!
CMD streamlit run app.py