#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
using namespace std;
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
int op, op2;
int ask(int ty, int l, int r)
{
    cout << ty << " " << l << " " << r << flush << endl;
    int res;
    cin >> res;
    return res;
}
void tell(int l, int r)
{
    cout << "! " << l << " " << r << flush << endl;
}
void Murasame()
{
    int n;
    cin >> n;
    if (n == 1)
    {
        tell(1, 1);
        return;
    }
    int sum = ask(2, 1, n) - n * (n + 1) / 2;
    int l = 1, r = n;
    while (l < r)
    {
        int mid = (l + r) / 2;
        if (ask(1, 1, mid) < ask(2, 1, mid))
            r = mid;
        else
            l = mid + 1;
    }
    tell(l, l + sum - 1);
}
signed main()
{
    //	ios::sync_with_stdio(false);
    //	cin.tie(0);
    //	cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}