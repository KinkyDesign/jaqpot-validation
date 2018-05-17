FROM python:2

MAINTAINER Pantelis Karatzas <pantelispanka@gmail.com>

ADD requirements.txt /jaqpot-validation/requirements.txt
ADD valid_service.py /jaqpot-validation/valid_service.py

RUN pip install -r /jaqpot-validation/requirements.txt

EXPOSE 5000

CMD ["python","/jaqpot-validation/valid_service.py"]
