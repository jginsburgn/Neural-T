echo "[" > samples;
for i in $(seq 1 $1); do python2.7 judge.py | python2.7 build_sample.py >> samples; done
echo "]" >> samples;
