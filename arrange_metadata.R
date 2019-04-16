library(dplyr)
library(readr)
library(quanteda)
library(lubridate)

R = read_rds("data/RedditExeter.rds") %>% mutate(month = month(date))
R


R$embedded_media = "NONE"
  R$embedded_media[!is.na(R$media_type)] = "other"
  R$embedded_media[grepl("clyp|dailymotion|liveleak|ted|twitch|vid.me|vimeo|vine|youtube", R$media_type)] = "video"
  R$embedded_media[grepl("flickr|funny|gfycat|gifs|giphy|gph|imgur|instagram", R$media_type)] = "img_gif"
  table(R$embedded_media)

R$self_text = "no"
  R$self_text[R$selftext!=""] = "yes"
  R$self_text[R$selftext=="[deleted]"] = "deleted"
  R$self_text[R$selftext=="[removed]"] = "removed"  
  
table(R$self_text)  
table(R$self_text, R$is_self)  
table(R$stickied)
table(R$over_18, R$month)
table(R$archived)


R2 = R %>% transmute(subreddit, author, 
                     date, month = month(date, label = TRUE) %>% factor %>% factor(labels = c("Apr","May","Jun","Jul","Aug","Sep")), 
                           day = day(datetime), hour = hour(datetime)+minute(datetime)/60,
                     title, domain = ifelse(grepl("self", domain), "self", domain), 
                     num_comments, is_self, self_text, score, logscore = log(score+1),
                     is_troll) %>% 
           mutate(subreddit = factor(subreddit), 
                  is_self = factor(is_self))
summary(R2)

add is troll, category

R2 %>% group_by(domain) %>% summarise(n=n()) %>% arrange(desc(n)) %>% View

x = R2$domain
t = tokens(x) %>% tokens_split(separator = ".") %>% tokens_replace("youtu", "youtube")
d = dfm(t)
topfeatures(d, 20)
d2 = d %>% dfm_remove(c("com","co","uk","org","net","us","it","in","be","m","i")) 
           
topfeatures(d2,20)

library(magrittr)

for (m in levels(R2$month))
{
print(m)
R2 %>% filter(month == m, is_troll) %$% table(author) %>% print
cat("=====\n\n")
}

