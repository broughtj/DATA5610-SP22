BEGIN { FS=","; OFS="," }

NR == 1 { next }

{ if ($1 >= begdate)  print $1, $3 }
