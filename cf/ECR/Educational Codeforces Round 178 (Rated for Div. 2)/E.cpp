#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
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
int n, k;
string s, t;
vi g[26];
int a[1000005];
int dp[1000005];
void f()
{
    dp[n] = 0;
    ff(i, 0, n - 1) g[s[i] - 'a'].pb(i);
    ff(i, 0, n - 1) dp[i] = inf;
    ff(i, 0, k - 1) a[i] = n;
    ffg(i, n - 1, 0)
    {
        ff(j, 0, k)
        {
            dp[i] = min(dp[i], dp[a[j]] + 1);
        }
        a[s[i] - 'a'] = i;
    }
}
void Murasame()
{
    cin >> n >> k;
    cin >> s;
    f();
    int q;
    cin >> q;
    while (q--)
    {
        cin >> t;
        int pre = -1;
        ff(i, 0, t.size() - 1)
        {
            int x = upper_bound(all(g[t[i] - 'a']), pre) - g[t[i] - 'a'].begin();
            if (x == g[t[i] - 'a'].size())
            {
                pre = n;
                break;
            }
            else
                pre = g[t[i] - 'a'][x];
        }
        cout << dp[pre] << endl;
    }
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
        Murasame();
    }
    return 0;
}