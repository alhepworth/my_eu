LIFE_OUTPUTS := \
	life/output/life.pkl.gz
OUTPUTS += $(CREATIVE_OUTPUTS)

life/life_clean.ipynb.html: life/life_clean.ipynb life/input/**/* postcodes/input/**/*
$(CREATIVE_OUTPUTS): life/life_clean.ipynb.html
