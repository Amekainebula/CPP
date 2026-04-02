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
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
void Murasame()
{
    int n, t;
    cin >> n >> t;
    string s;
    cin >> s;
    int mx = 0;
    int cnt = 0;
    vi pre;
    ff(i, 0, n - 1)
    {
        if (s[i] == '1')
        {
            if (cnt)
            {
                pre.pb(cnt);
                cnt = 0;
            }
            pre.pb(-1);
        }
        else
        {
            cnt++;
        }
        if (i == n - 1 && s[i] == '0')
        {
            pre.pb(cnt);
        }
    }
    if (pre[0] != -1 && pre[pre.size() - 1] != -1)
    {
        pre[0] += pre[pre.size() - 1];
        pre.pop_back();
    }
    bool ok = 1;
    for (auto x : pre)
    {
        mx = max(mx, x);
    }
    int ans = max(max(0LL, mx - 1 - t), mx - 2 * t);
    for (auto x : pre)
    {
        if (ok && x == mx)
        {
            ok = 0;
            continue;
        }
        ans += max(0LL, x - 2 * t);
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
        Murasame();
    }
    return 0;
}