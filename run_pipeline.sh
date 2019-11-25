#!/bin/bash

# WARNING: I've changed a lot of stuff from the original (Sander's folder on pataphysique). Will not run.

echo "Starting full pipeline run"
echo "Step 1: Running LinesHandler"

fName=news_mini.json
oName1=predArgs_tense_gen.txt
oName2=predArgs_tense_NE.txt
echo "0" > offset.txt
i=0
echo $i
while [ -f offset.txt ]; do
if [ $i -eq 0 ]
then
        java -cp lib/*:bin  entailment.LinesHandler $fName $i $oName1 $oName2 1>tmpo.txt 2>lineNumbers_b.txt
else
        java -cp lib/*:bin  entailment.LinesHandler $fName $i $oName1 $oName2 1>>tmpo.txt 2>>lineNumbers_b.txt
fi
i=$[ $i + 1 ]
echo $i
done

echo "Step 2: Running Util, PredArgsToJson"

java -Xmx10G -cp lib/*:bin entailment.Util $oName1 true true 12000000 news_linked.json 1> news_tense_gen.json

echo "Step 3: Running EntailGraphFactoryAggregator"

java -Xmx10G -cp lib/*:bin  entailment.vector.EntailGraphFactoryAggregator

