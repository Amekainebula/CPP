#include <fstream>
#include <vector>
using namespace std;

void getNext(const string &pattern, vector<int> &next)
{
    int m = pattern.size();
    next.resize(m);
    int j = 0;
    for (int i = 1; i < m; ++i)
    {
        while (j > 0 && pattern[i] != pattern[j])
            j = next[j - 1];
        if (pattern[i] == pattern[j])
            ++j;
        next[i] = j;
    }
}

bool kmpSearch(const string &text, const string &pattern)
{
    int n = text.size();
    int m = pattern.size();
    vector<int> next;
    getNext(pattern, next);
    int j = 0;
    for (int i = 0; i < n; ++i)
    {
        while (j > 0 && text[i] != pattern[j])
            j = next[j - 1];
        if (text[i] == pattern[j])
            ++j;
        if (j == m)
        {
            return 1;
            // j = next[j - 1];
        }
    }
    return 0;
}

bool is_palindrome(const string &s)
{
    int n = s.size();
    for (int i = 0; i < n / 2; i++)
    {
        if (s[i] != s[n - 1 - i])
        {
            return 0;
        }
    }
    return 1;
}

bool same_sum(const string &s, const string &t)
{
    long long ss = 0, st = 0;
    for (int i = 0; i < s.size(); i++)
    {
        ss += s[i] - 'a' + 1;
    }
    for (int i = 0; i < t.size(); i++)
    {
        st += t[i] - 'a' + 1;
    }
    return ss == st;
}

bool all_small(const string &s)
{
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] < 'a' || s[i] > 'z')
        {
            return 0;
        }
    }
    return 1;
}
int main()
{
    ifstream input("input");             // 打开输入文件
    ifstream user_output("user_output"); // 打开输出文件
    int T;
    if (!(input >> T))
        return 1;
    for (int i = 1; i <= T; i++)
    {
        string s, t;
        if (!(input >> s))
            break;
        if (!(user_output >> t))
        {
            return 1;
        }

        if (t.size() <= s.size() * 2 && all_small(t) && same_sum(s, t) && is_palindrome(t) && kmpSearch(t, "promise"))
        {

        }
        else
        {
            return 0;
        }
    }
    return 1;
}