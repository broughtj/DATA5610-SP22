#0 to run: awk -f negprc.awk WaterStocks.csv

#1 Set input/output sep and print headers
BEGIN {
  FS = ","
  OFS = " "
  print "COUNT", "PERMNO", "DATE", "PRICE", "ABSPRC"
}

#2 Skip header row
NR == 1 { next }

#3 Print rows with negative prices
$8 < 0 { print ++row, $1, $2, $8, sqrt($8^2) }
