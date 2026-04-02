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
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1);
    ff(i, 1, n) cin >> a[i];
    string s;
    cin >> s;
    vi d1(1, 0), d2(1, 0);
    ff(i, 1, n)
    {
        if (s[i - 1] == 'R')
            d1.pb(a[i]);
        else
            d2.pb(a[i]);
    }
    sort(all1(d1), greater<int>());
    sort(all1(d2), greater<int>());
    int ans = 0;
    int a1 = min(d1.size() - 1, d2.size());
    int a2 = min(a1, (int)d2.size() - 1);
    ff(i, 1, a1) ans += d1[i];
    ff(i, 1, a2) ans += d2[i];
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