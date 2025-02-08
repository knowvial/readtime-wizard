# readtime-wizard
ML Model hosted in docker - learning example

## Instructions:
```sh
git clone https://github.com/knowvial/readtime-wizard.git

cd readtime-wizard

docker build -t readtime-wizard .

docker run -d -p 80:80 readtime-wizard
```

## Sample data:
```
Book Title: The Great Gatsby
Pages: 180
Reading Level: Intermediate
Genre: fiction
Available Time: 60 (minutes per day)
Reading Speed: Medium
```

```
Book Title: Learning Python Programming
Pages: 450
Reading Level: Beginner (1)
Genre: technical
Available Time: 45 (minutes per day)
Reading Speed: Slow (1)
```

```
Book Title: Introduction to Psychology
Pages: 600
Reading Level: Intermediate (2)
Genre: textbook
Available Time: 90 (minutes per day)
Reading Speed: Medium (2)
```