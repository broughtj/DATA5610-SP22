all:
	Rscript -e 'rmarkdown::render("chapter04.Rmd")'


pdf:
	Rscript -e 'rmarkdown::render("chapter04.Rmd", output_format="beamer_presentation")'


iochapter04:
	Rscript -e 'rmarkdown::render("chapter04.Rmd", output_format="iochapter04_presentation")'

slidy:
	Rscript -e 'rmarkdown::render("chapter04.Rmd", output_format="slidy_presentation")'


clean:
	rm chapter04.pdf chapter04.html
