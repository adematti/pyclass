DEST?=_inst
URL?=https://github.com/adematti/class_public/archive/refs/tags/v3.2.1b.tar.gz
#URL?=

TARBALL=tmp-class.tar.gz
UNPACK=tmp-class
SRC=$(UNPACK)

$(TARBALL):
	curl -fsSL $(URL) > $(TARBALL)
	#wget -O $(TARBALL) $(URL)

$(SRC): $(TARBALL)
	mkdir -p $(UNPACK)
	gzip -dc $(TARBALL) | tar xf - -C $(UNPACK)
	mv $(UNPACK)/class*/* $(UNPACK)/
	touch $@

$(SRC)/libclass.a: $(SRC)
	cd $(SRC); make libclass.a

$(DEST)/lib/libclass.a: $(SRC)/libclass.a
	mkdir -p $(DEST)
	cp -r $(SRC)/include $(DEST)
	cp -r $(SRC)/external $(DEST)
	mkdir -p $(DEST)/lib
	cp $(SRC)/libclass.a $(DEST)/lib
	mkdir -p $(DEST)/data
	cp $(SRC)/*.ini $(SRC)/*.pre $(DEST)/data/
	rm -rf $(TARBALL) $(UNPACK) $(SRC)

build: $(SRC)/libclass.a
install: $(DEST)/lib/libclass.a
clean:
	rm -rf $(DEST)
