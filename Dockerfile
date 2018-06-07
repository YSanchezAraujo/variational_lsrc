FROM rocker/shiny

RUN apt-get -yqq update \
    && apt-get -yqq install --no-install-recommends libhdf5-dev python3-dev \
    && R -e 'install.packages(c("h5", "varbvs"))' \
    && wget -O- https://bootstrap.pypa.io/get-pip.py | python3 - --no-cache-dir \
    && pip3 --no-cache-dir install numpy scipy pandas ipython nipype jupyter scikit-learn tensorflow==1.6 edward \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["/bin/bash"]
