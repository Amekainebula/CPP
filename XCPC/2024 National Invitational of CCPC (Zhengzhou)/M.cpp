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
int n;
vi a(300005 + 1), b(300005 + 1);
int ans = 0;
bool check(int mid)
{
    int l = -INF, r = INF;
    ff(i, 1, n)
    {
        l = max(l, a[i] - mid * b[i]);
        r = min(r, a[i] + mid * b[i]);
        if (l > r)
            return false;
    }
    //cout << mid << " " << l << " " << r << endl;
    ans = mid;
    return true;
}
void Murasame()
{
    cin >> n;
    ff(i, 1, n) cin >> a[i];
    ff(i, 1, n) cin >> b[i];
    int l = 0, r = 1e10;
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid))
            r = mid - 1;
        else
            l = mid;
    }
    if (check(l))
        cout << l << endl;
    else
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