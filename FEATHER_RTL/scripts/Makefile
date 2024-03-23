SHELL := /bin/csh

all:
	rm ../outputs/*
	rm ../reports/*
	rm ../rpt.tar.gz 
run:
	source :run_syn

del_rtl:
	rm ../RTL/*
comp:
	cp ../outputs/*.g.v  ../reports/
	cp ../outputs/*.sdc ../reports/
	tar -zcvf ../rpt.tar.gz ../reports
