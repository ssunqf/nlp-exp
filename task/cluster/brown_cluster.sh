#!/usr/bin/env bash

git clone https://github.com/percyliang/brown-cluster.git
cd brown-cluster
make
./wcluster --html2text ../cn.char.tok --min-occur 10 --threads 5