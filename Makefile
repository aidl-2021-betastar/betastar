docker:
	docker build -t codegram/betastar:latest .

sc2:
	docker build -f docker/Dockerfile.sc2 -t codegram/sc2:latest -t codegram/sc2:4.10 docker/
	docker push --all-tags codegram/sc2:latest
