{
  "query": {
    "nested": {
      "path": "skills",
      "query": {
        "bool": {
          "should": [
            {
              "bool": {
                "filter": [
                  {
                    "script": {
                      "script": {
                        "source": "doc['skills.similarity'].value > params.A + params.B * doc['skills.sentence_length'].value",
                        "params": {
                          "A": 0.5,
                          "B": 0.1
                        }
                      }
                    }
                  },
                  {
                    "term": {
                      "skills.canonical_skill_id.keyword": {
                        "value": "0344"
                      }
                    }
                  }
                ]
              }
            },
            {
              "bool": {
                "filter": [
                  {
                    "script": {
                      "script": {
                        "source": "doc['skills.similarity'].value <= params.A + params.B * doc['skills.sentence_length'].value",
                        "params": {
                          "A": 0.5,
                          "B": 0.1
                        }
                      }
                    }
                  },
                  {
                    "term": {
                      "skills.original_skill.keyword": {
                        "value": "python"
                      }
                    }
                  }
                ]
              }
            }
          ]
        }
      },
      "inner_hits": {}
    }
  }
}
