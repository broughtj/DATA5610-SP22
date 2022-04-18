awk -F "," 'NR == 1 { next } { print $1 }' WaterStocks.csv | sort | uniq > permnos.txt
