DEST?=_inst
PATCH?=foo
INCLUDE?=include

TARBALL?=tmp-class.tar.gz
UNPACK=tmp-class
SRC=$(UNPACK)

$(SRC): $(TARBALL)
	mkdir -p $(UNPACK)
	gzip -dc $(TARBALL) | tar xf - -C $(UNPACK)
	touch $@
	mv $(UNPACK)/*/* $(UNPACK)/
	[ -d $(PATCH) ] && cp -r $(PATCH)/* $(UNPACK)/

$(SRC)/libclass.a: $(SRC)
	cd $(SRC); make libclass.a

$(DEST)/lib/libclass.a: $(SRC)/libclass.a
	mkdir -p $(DEST)
	for inc in $(INCLUDE) ; do \
	  cp -r $(SRC)/$$inc $(DEST) ; \
	done
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
