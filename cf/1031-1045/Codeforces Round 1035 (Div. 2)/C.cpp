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
    int n, l, r, k;
    cin >> n >> l >> r >> k;
    if (n == 2)
    {
        cout << -1 << endl;
        return;
    }
    if (n % 2 == 1)
    {
        cout << l << endl;
        return;
    }
    int t = log2(l);
    t++;
    t = pow(2, t);
    if (t > r)
    {
        cout << -1 << endl;
        return;
    }
    if (k == n || k == n - 1)
    {
        cout << t << endl;
    }
    else
    {
        cout << l << endl;
    }
    // 0110 6
    // 0111 7
    // 1000 8
    // int a[8] = {2, 2, 2, 2, 2, 2, 4, 4};
    // int b[6] = {4, 4, 4, 4, 8, 8};
    // cout << (b[0] & b[1] & b[2] & b[3] & b[4] & b[5]) << endl;
    // cout << (b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5]) << endl;
    //     cout << (a[0] & a[1] & a[2] & a[3] & a[4] & a[5] & a[6] & a[7]) << endl;
    //     cout << (a[0] ^ a[1] ^ a[2] ^ a[3] ^ a[4] ^ a[5] ^ a[6] ^ a[7]) << endl;
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