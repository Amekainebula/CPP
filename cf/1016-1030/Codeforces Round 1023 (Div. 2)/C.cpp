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
const int N = 2e5 + 5;
void Murasame()
{
    int n, k;
    cin >> n >> k;
    string s;
    cin >> s;
    s = '1' + s;
    vi a(n + 1);
    int ok = 0;
    ff(i, 1, n)
    {
        cin >> a[i];
        if (s[i] == '0')
        {
            ok = i;
            a[i] = -1e13;
        }
    }
    vi dp(n + 1, 0);
    int now = -1e13;
    ff(i, 1, n)
    {
        dp[i] = max(dp[i - 1] + a[i], a[i]);
        now = max(now, dp[i]);
    }
    if (now > k || (!ok && now != k))
    {
        cout << "NO\n";
        return;
    }
    cout << "YES\n";
    if (now == k)
    {
        ff(i, 1, n)
        {
            cout << a[i] << " \n"[i == n];
        }
    }
    else
    {
        int mx = 0;
        int now = 0;
        int L, R;
        ff(i, ok + 1, n)
        {
            now += a[i];
            mx = max(mx, now);
        }
        L = mx;
        mx = 0;
        now = 0;
        ffg(i, ok - 1, 1)
        {
            now += a[i];
            mx = max(mx, now);
        }
        R = mx;
        a[ok] = k - L - R;
        ff(i, 1, n)
        {
            cout << a[i] << " \n"[i == n];
        }
    }
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