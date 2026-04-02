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
int kmp(string s, string t)
{
    int n = s.size(), m = t.size();
    vector<int> pi(m + n);
    int sum = 0;
    for (int i = 0, j = 0; i < n; i++)
    {
        while (j && t[j] != s[i])
            j = pi[j - 1];
        if (t[j] == s[i])
            j++;
        if (j == m)
        {
            sum++;
            j = pi[j - 1];
        }
    }
    return sum;
}
void solve()
{
    int n, m;
    cin >> n >> m;
    int x;
    string s = "", t = "";
    ff(i, 1, n)
    {
        cin >> x;
        char c = x + '0';
        s += c;
    }
    ff(i, 1, m)
    {
        cin >> x;
        char c = x + '0';
        t += c;
    }
    int ans = kmp(s, t);
    cout << (ans >= 2 ? "Yes" : "No") << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}