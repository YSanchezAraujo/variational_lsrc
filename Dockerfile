FROM rocker/shiny

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get -yqq update \
    && apt-get -yqq install --no-install-recommends libhdf5-dev libssl-dev libssh2-1-dev python3-dev \
    && R -e 'install.packages("h5"); install.packages("devtools", repos = "http://cran.us.r-project.org"); require(devtools); install_github("pcarbo/varbvs",subdir = "varbvs-R")' \
    && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3 - --no-cache-dir \
    && pip3 --no-cache-dir install numpy scipy pandas ipython nipype jupyter scikit-learn tensorflow==1.6 edward \
    && wget https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz \
    && tar -xvf julia-1.1.0-linux-x86_64.tar.gz && rm *.tar.gz && mv julia* julia \
    && mv julia /opt && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && JULIA_PKGDIR=/opt/julia \
    && echo 'using Pkg;packs=["Distributions", "Conda", "CSV", "DataFrames", "ForwardDiff", "PyCall", "GLM", "LinearAlgebra", "Turing", "MCMCChain", "Plots", "StatsPlots"];for i in packs;Pkg.add(i);end' >> packs.jl \
    && julia packs.jl && echo 'using Conda;Conda.add("scipy");' >> inst_scipy.jl && julia inst_scipy.jl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["/bin/bash"]
