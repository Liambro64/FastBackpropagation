#include <curl/curl.h>

int main(int argc, char *argv[])
{
  CURLcode ret;
  CURL *hnd;
  struct curl_slist *headers;

  headers = NULL;
  headers = curl_slist_append(headers, "accept: text/event-stream");
  headers = curl_slist_append(headers, "accept-language: en-GB,en-US;q=0.9,en;q=0.8");
  headers = curl_slist_append(headers, "cache-control: no-cache");
  headers = curl_slist_append(headers, "origin: https://app.plus500.com");
  headers = curl_slist_append(headers, "priority: u=1, i");
  headers = curl_slist_append(headers, "sec-ch-ua: \"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"");
  headers = curl_slist_append(headers, "sec-ch-ua-mobile: ?0");
  headers = curl_slist_append(headers, "sec-ch-ua-platform: \"Windows\"");
  headers = curl_slist_append(headers, "sec-fetch-dest: empty");
  headers = curl_slist_append(headers, "sec-fetch-mode: cors");
  headers = curl_slist_append(headers, "sec-fetch-site: same-site");
  headers = curl_slist_append(headers, "cookie: webvisitid=b97fc62c-3087-481f-b08b-c8e63fa42e98; AMP_MKTG_8f1ede8e9c=JTdCJTdE; __cf_bm=UUEhFKTXS7_K.4AfBXqGk_cmIAUKUTwwYELaST4FYM4-1752145701-1.0.1.1-LCFVjk3O1OpWObmHAtUI8.uaTvWiw83D7tBzzqEwQonGNvVDs.lhIeG3sJyxg_2fTmW6KOHeJhJwWnKfITzcErBpcQoknqfdi9QTDdrUSVo; _cfuvid=RBW.vA209A_Kt8ckxN27qfAteKgw.tWdBUonDZpLfCI-1752145701001-0.0.1.1-604800000; cf_clearance=BqSlRTh.RHJf24sszzgg6flsb.SDfKxBgyKklH7Nz_4-1752145701-1.2.1.1-rFO4rkIreRKar838Q7W6GHwBX0W8TaI.YI7V4KV82rCKUhXdHVX9xqiRF_md4SV.PlOrcuMQaAkzUiAAE0KrPjx4CgJDZpL0FHO9l5rDmd4ujxOpdw0GXJzXJBzDyrEViNXD.zn5L7A1eN.aMtA2Ggkbl4mXGAAyaBdOmpsWiNnxNDc528XerZHZWebtxHV2h_NYcNygOn9IbAOFbL3_uebEty56Y5Exi6J36E2WzjU; X-Origin-IFE=Qk9WC9sy0z; IP=!Lx9nX3ZiNIDcE8PzV6e823ydVOUkxv+HODtzfpKIkAgW1GXDdQSNDCt95v3qXcCdVwd2JFWK9RhrtUiL76ekKu4clN3T2croiPfNRbbmI7Az99KT3aF900U4QAbu/+01gmYnd1Q2WFsyqFNw0dYraQeWccUbYqiWA2GU3HE1RO6UZt3Xqd8scWtqusbfwsVHgBs+8OjqtzxgQuzCPsZ2aNiKvtfJ3n5dE3hG7462i3BHcP8qSxZ0hoIa9Iio47j8jql98wgraQ==; AMP_8f1ede8e9c=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjJhOGMyYWUxNi01ZmM3LTQ4MGQtYThhYy0yODFlNTBiYjJmYTglMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzUyMTQ1NzAxMTU4JTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTc1MjE0NTcwODcyOSUyQyUyMmxhc3RFdmVudElkJTIyJTNBMTc4JTdE");

  hnd = curl_easy_init();
  curl_easy_setopt(hnd, CURLOPT_BUFFERSIZE, 102400L);
  curl_easy_setopt(hnd, CURLOPT_URL, "https://api.plus500.com/signalr/hubs/reconnect?transport=serverSentEvents&messageId=d-10711F03-B%2C0%7CDKFm%2C12%7CDKFn%2C1&clientProtocol=1.5&WebTraderServiceId=becdda84-0a8f-4d24-9d30-3c02156c5143&UserSessionId=c93182d4-577f-47d3-8d65-dfcd378acf8f&Hash=0e6faad1a8b118dd1dee94c5523c9ee3ce0b9f7fcbbb4d23634a77efded67b21&ClientType=WebAppDesktop&BlockForexOnSunday=False&connectionToken=Qnb26qzsLW2DZ3SgMnlH88c86HVF19gsyQajlgMJ6EDzBNKaWL9aOuGhiPhZFuGKl6U6SnS72HZJU6tyBHCfKjLvDhGFbUSbXRMvqrYQ9B4wo9EXWVwdH3Aqh3%2B9e3s7&connectionData=%5B%7B%22name%22%3A%22c%22%7D%5D&tid=9");
  curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
  curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(hnd, CURLOPT_REFERER, "https://app.plus500.com/");
  curl_easy_setopt(hnd, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36");
  curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
  curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
  curl_easy_setopt(hnd, CURLOPT_FTP_SKIP_PASV_IP, 1L);
  curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);

  ret = curl_easy_perform(hnd);

  curl_easy_cleanup(hnd);
  hnd = NULL;
  curl_slist_free_all(headers);
  headers = NULL;

  return (int)ret;
}
