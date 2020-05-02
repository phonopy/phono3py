#!/bin/bash

#br=`git branch |tail -n1`
br=`git branch|grep '*'|sed s/\*//`
echo $br
read o u f <<< `git remote -v |grep origin |grep fetch`
echo "Repo: $o $u $br"
git describe --tags --dirty
TD=`mktemp -d`
WD=`pwd`
git branch
git clone $u $TD
git checkout $br
cd $TD
echo $u $br
git describe --tags --dirty | sed -e 's/\([.0-9]*\)-\(.*\)-g.*/\2/' -e 's/^[vr]//g' -e 's/rc-//g' > $WD/__nanoversion__.txt
cd $WD
rm -rf "$TD"
