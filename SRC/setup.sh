#!/bin/bash

rm -rf Match-2.3.1
echo "*** Getting codes ***"
wget -q -O - 'http://www.lorraine-lisiecki.com/match-2.3.1.tgz' | tar xvzf - | awk '{print " ",$0}'
(
	cd Match-2.3.1/ || exit
	echo "*** Patching ***"
	cat ../patch.gcc-4.9 | patch -p1 | awk '{print " ",$0}'
	cd Match/ || exit
	echo "*** Compiling ***"
	make clean | awk '{print " ",$0}'
	make | awk '{print " ",$0}'
	[ -f match ] && echo "File will be installed on 'match'"

	echo ""
	echo "*** Done ***"
	echo ""
	cp -v match ../../../
	cd ../Examples/
	cp LR04core b806l ../../../
)
rm -rf Match-2.3.1


