#1 Set the input and output field separators and print headers
BEGIN { 
  FS=","; OFS="|";
  print "PERMNO", "DATE", "PRC"
}

#2 Skip the header row (not really needed)
NR == 1 { next }

#3 Match the command line permno input variable
# Take the absolute value of price by sqrt(x^2)
$1 ~ permno {print $1, $2, sqrt($8^2) }
