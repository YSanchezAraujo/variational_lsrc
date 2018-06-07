FROM rocker/shiny

WORKDIR /tmp

RUN wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.1.tar \
    && tar xf hdf5-1.10.1.tar

WORKDIR /tmp/hdf5-1.10.1

RUN ./configure --prefix=/usr/local --enable-cxx --enable-build-mode=production \
    && make -j 16 \
    && make install \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/h5.conf \
    && ldconfig

RUN R -e 'install.packages("h5")'
RUN R -e 'install.packages("varbvs")'

RUN apt-get -yqq update \
    && apt-get -yqq install python3-pip python3-dev \
    && pip3 --no-cache-dir install numpy scipy pandas ipython nipype jupyter scikit-learn tensorflow==1.6 edward

ENTRYPOINT ["/bin/bash"]
