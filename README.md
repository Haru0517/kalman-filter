# python-docker-template
Python + Docker environment template.

## Requirement

- [Docker](https://www.docker.com/)
  - docker-compose

## Usage
Build image:

```console
$ cp docker/python/.env.example docker/python/.env
$ docker-compose build
```

Run container:

```console
$ docker-compose up -d
$ docker-compose exec python bash
```

Go to http://localhost:8888 and you'll see the notebooks.




## Licence

This software is released under the MIT License, see [LICENSE](https://github.com/Haru0517/python-docker-template/blob/master/LICENSE).

## Author

[Yuto Chikazawa](https://github.com/Haru0517)