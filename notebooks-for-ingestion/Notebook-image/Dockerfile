FROM quay.io/modh/odh-generic-data-science-notebook:v2-20240424

RUN pip install pip==22.3.1 setuptools==65.3.0

USER 0
RUN yum -y install unixODBC && \
    yum clean all

USER 1001

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    chmod -R g+w /opt/app-root/lib/python3.9/site-packages && \
    fix-permissions /opt/app-root -P

USER 0

RUN python -m nltk.downloader -d /usr/lib/nltk_data all
USER 1001