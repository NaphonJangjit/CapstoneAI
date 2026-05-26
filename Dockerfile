FROM maven:3-eclipse-temurin-26 AS build
WORKDIR /build
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src/ src/
RUN mvn package -DskipTests && \
    mvn dependency:copy-dependencies -DoutputDirectory=target/dependency

FROM eclipse-temurin:26-jre-noble
WORKDIR /app
COPY --from=build /build/target/dependency/ /app/lib/
COPY --from=build /build/target/CapstoneAI-1.0-SNAPSHOT.jar app.jar

EXPOSE 9922

ENV CAPSTONE_HOST=0.0.0.0
ENV CAPSTONE_PORT=9922
ENV SOLR_URL=http://host.docker.internal:8983/solr/chapters

ENTRYPOINT ["java", "-cp", "app.jar:/app/lib/*", "naphon.capstone.ai.Main"]
