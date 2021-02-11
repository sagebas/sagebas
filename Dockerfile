FROM python:3

WORKDIR C:\\Users\\robore\\BAS_Files\\Research\\Code\\SAGE\\Dockers\\GGF_realtime_forecast

RUN git clone https://github.com/robert-m-shore/GGF_realtime_forecast ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./GGF_RTF.py" ]

