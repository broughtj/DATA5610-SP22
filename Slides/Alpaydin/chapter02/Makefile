all:
	Rscript -e 'rmarkdown::render("chapter02.Rmd")'


pdf:
	Rscript -e 'rmarkdown::render("chapter02.Rmd", output_format="beamer_presentation")'


iochapter02:
	Rscript -e 'rmarkdown::render("chapter02.Rmd", output_format="iochapter02_presentation")'

slidy:
	Rscript -e 'rmarkdown::render("chapter02.Rmd", output_format="slidy_presentation")'


clean:
	rm chapter02.pdf chapter02.html
