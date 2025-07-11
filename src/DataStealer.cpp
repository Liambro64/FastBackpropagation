#include "../Project.hpp"

size_t base_callback(char *ptr, size_t size, size_t nmemb, void *userdata)
{
	std::string s(ptr, size * nmemb);
	std::cout << s <<std::endl;
	return size * nmemb;
}
DataStealer::DataStealer(size_t (*callback)(char *, size_t, size_t, void *)) {
	if (callback == nullptr) {
		this->callback = &base_callback;
	} else {
		this->callback = callback;
	}
}
std::pair<std::string, std::string> DataStealer::LoadFromFile(std::string infile, std::string rtfile)
{
	std::string cmd = "cat " + infile + " | curlconverter --language c - > " + rtfile;
	system(cmd.c_str());
	std::fstream cmdFile(rtfile);
	std::string line;
	std::string cookieLine;
	std::string urlLine;
	line.resize(1024);
	int i = 0;
	while (std::getline(cmdFile, line))
	{
		if (line.find("cookie: ") != std::string::npos)
		{
			cookieLine = line.substr(40);
		}
		else if (line.find("CURLOPT_URL") != std::string::npos)
		{
			urlLine = line.substr(38);
		}
	}
	size_t end = cookieLine.find("\");");
	size_t end2 = urlLine.find("\");");
	cookieLine = cookieLine.substr(0, end);
	urlLine = urlLine.substr(0, end2);
	return { cookieLine, urlLine };
}
void DataStealer::ConnectToStream(CURLcode *ret, std::string file)
{
	CURL *hnd;
	struct curl_slist *headers;
	auto [cookieLine, urlLine] = LoadFromFile(file);

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
	char *cookie = (char *)malloc(cookieLine.size() + 1);
	std::copy(cookieLine.begin(), cookieLine.end(), cookie);
	cookie[cookieLine.size()] = '\0';
	headers = curl_slist_append(headers, cookie);

	hnd = curl_easy_init();
	curl_easy_setopt(hnd, CURLOPT_BUFFERSIZE, 102400L);
	char *url = (char *)malloc(urlLine.size() + 1);
	std::copy(urlLine.begin(), urlLine.end(), url);
	url[urlLine.size()] = '\0';
	curl_easy_setopt(hnd, CURLOPT_URL, url);
	curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
	curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(hnd, CURLOPT_REFERER, "https://app.plus500.com/");
	curl_easy_setopt(hnd, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36");
	curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
	curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
	curl_easy_setopt(hnd, CURLOPT_FTP_SKIP_PASV_IP, 1L);
	curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);
	curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, callback);

	curl_easy_perform(hnd);

	curl_easy_cleanup(hnd);
	hnd = NULL;
	curl_slist_free_all(headers);
	headers = NULL;
}