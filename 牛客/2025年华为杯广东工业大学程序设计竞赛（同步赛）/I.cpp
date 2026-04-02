#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
// 计算部分匹配表（next数组）
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
// KMP算法搜索匹配串出现的所有位置
vector<int> kmpSearch(const string &text, const string &pattern)
{
    int n = text.size();
    int m = pattern.size();
    vector<int> next;
    getNext(pattern, next);
    vector<int> positions;
    int j = 0;
    for (int i = 0; i < n; ++i)
    {
        while (j > 0 && text[i] != pattern[j])
            j = next[j - 1];
        if (text[i] == pattern[j])
            ++j;
        if (j == m)
        {
            positions.push_back(i - m + 1);
            j = next[j - 1];
        }
    }
    return positions;
}
void solve()
{
    int n, m;
    cin >> n >> m;
    string s, t;
    cin >> s >> t;
    vector<int> positions = kmpSearch(s, t);
    sort(all(positions));
    int ans = 0;
    int now = -1;
    for (auto x : positions)
    {
        if (x > now)
        {
            ans++;
            now = x + t.size() - 1;
        }
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}